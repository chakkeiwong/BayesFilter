"""Progress continuation, dynamic pilot dependencies and bounded settlement."""
from dataclasses import replace
from types import SimpleNamespace
import time

import pytest

from bayesfilter.testing.inference_validation import campaign_recovery as recovery
from bayesfilter.testing.inference_validation.campaign_pool import run_pool
from bayesfilter.testing.inference_validation.storage import read_json, write_json, file_hash
from scripts import run_hmc_ssm_pooled_campaign as campaign
from tests.inference_validation.test_campaign_recovery import saved_fit, stopped
from tests.inference_validation.test_ssm_priced_main import inventory


def test_productive_job_can_finish_after_three_turns_and_keeps_fairness():
    now, calls = [0.], []
    def execute(job, allowance, deadline, attempt):
        calls.append((job['job_id'], attempt))
        now[0] += 1.
        return ({'status': 'complete', 'posterior_passed': False}
                if job['job_id'] == 'peer' or attempt == 5 else stopped(1.))
    result = run_pool([{'job_id': j, 'case': 'K0'} for j in ['long', 'peer']], execute,
        quantum_seconds=1., deadline=8., max_attempts=None, clock=lambda: now[0])
    assert calls[:3] == [('long', 1), ('peer', 1), ('long', 2)]
    assert result['all_workloads_complete'] and len(calls) == 6
    assert result['rows'][1]['attempts'][0]['posterior_passed'] is False


@pytest.mark.parametrize('progress', [True, False])
def test_unlimited_attempt_policy_still_obeys_deadline_and_no_progress(progress):
    now = [0.]
    def execute(*args):
        now[0] += 1.
        return stopped(1., **({} if progress else {'progress': {}}))
    result = run_pool([{'job_id': 'fit', 'case': 'K2'}], execute,
        deadline=4.5, quantum_seconds=1., max_attempts=None, clock=lambda: now[0])
    row = result['rows'][0]
    assert len(row['attempts']) == (4 if progress else 1)
    assert row['status'] == ('incomplete_pool_deadline' if progress else 'incomplete_no_progress')
    assert now[0] <= 4.5


def test_broken_executor_cannot_spin_without_consuming_wall_time():
    with pytest.raises(ValueError, match='monotonic time'):
        run_pool([{'job_id': 'fit', 'case': 'K2'}], lambda *a: stopped(),
            deadline=10., quantum_seconds=1., max_attempts=None, clock=lambda: 0.)


def test_discovery_unlocks_main_even_when_last_pilot_empties_queue():
    now, calls = [0.], []
    pilot = {'job_id': 'pilot', 'case': 'K6'}
    main = {'job_id': 'main', 'case': 'K6'}
    def discover(rows):
        return [pilot, main] if rows and rows[0]['status'] == 'complete' else [pilot]
    def execute(job, *args):
        now[0] += 1.
        calls.append(job['job_id'])
        return {'status': 'complete', 'posterior_passed': False}
    result = run_pool([], execute, deadline=10., quantum_seconds=1., max_attempts=None,
        discover=discover, clock=lambda: now[0])
    assert calls == ['pilot', 'main']
    assert result['completed'] == 2


def test_discovery_rejects_changed_job_definition():
    n = [0]
    def discover(rows):
        n[0] += 1
        return [{'job_id': 'same', 'case': 'K0', 'cell': str(n[0])}]
    with pytest.raises(ValueError, match='definition changed'):
        run_pool([], lambda *a: {'status': 'complete'}, discover=discover,
            deadline=10., quantum_seconds=1., max_attempts=1, clock=lambda: 0.)


def test_real_recovery_boundary_receives_fifth_attempt_without_resetting_cost(tmp_path, design, monkeypatch):
    source, cell, d, identity = saved_fit(tmp_path, design)
    initial = recovery.immutable_fit_evidence(cell / 'replication-0000')
    calls = []
    def supervise(command, log, seconds, device, **kwargs):
        calls.append(seconds)
        prefix = str(log).removesuffix('.log')
        write_json(prefix + '-manifest.json', {'design_identity': d.identity,
            'source': {'identity': identity['source_identity']}, 'runtime': {}})
        if len(calls) == 5:
            write_json(cell / 'replication-0000/independent_assessment.json', {'posterior_passed': False})
            return {'status': 'complete', 'elapsed_seconds': .01, 'exit_code': 0}
        return stopped(.01)
    monkeypatch.setattr(recovery, 'supervise_fit', supervise)
    job = campaign.make_job('long', 'K2', cell, first_quantum=.2)
    result = run_pool([job], campaign.executor(source, tmp_path / 'attempts'),
        deadline=time.monotonic() + 10., quantum_seconds=.2, max_attempts=None)
    assert result['all_workloads_complete'] and len(calls) == 5
    last = result['rows'][0]['attempts'][-1]
    assert last['cumulative_fit_seconds'] == pytest.approx(10.05)
    assert all(file_hash(p) == sha for p, sha in initial.items())
    assert campaign.ValidationDesign.from_payload(read_json(cell / 'isolated_design.json')).identity == d.identity
    assert read_json(cell / 'replication-0000/independent_assessment.json')['posterior_passed'] is False


@pytest.fixture
def continuation_inventory(tmp_path, design, monkeypatch):
    prepared, previous, root = (tmp_path / p for p in ['prepared', 'previous', 'new'])
    monkeypatch.setattr(campaign, 'PREPARED', prepared)
    template, _ = inventory(design)
    write_json(prepared / 'main-unpriced.json', template)
    slots, states = [], {}
    for d in template['designs']:
        case, ident = d['options']['campaign_case'], d['design_id']
        cell = previous / 'main' / ident
        status = 'unpriced_workload'
        if case in {'K0', 'K2'} and ident.endswith(('slot-0', 'slot-1')):
            write_json(cell / 'isolated_design.json', d)
            status = 'repair_stage_incomplete' if case == 'K0' else 'incomplete_attempt_limit'
        slots.append({'design_id': ident, 'case': case, 'status': status, 'cell': str(cell)})
    complete = next(s for s in slots if s['case'] == 'K0')
    complete['status'] = 'preserved_complete'
    assessment = write_json(campaign.Path(complete['cell']) / 'replication-0000/independent_assessment.json',
                            {'posterior_passed': False})
    states[complete['cell']] = (True, True)
    for i in range(8):
        d = next(d for d in template['designs'] if d['options']['campaign_case'] == f'K{i}')
        cell = prepared / 'pricing' / f'price-K{i}'
        write_json(cell / 'isolated_design.json', d)
        states[str(cell)] = (i != 1, i != 1)
    pilot = {'job_id': 'pilot-K1', 'case': 'K1', 'cell': str(prepared / 'pricing/price-K1')}
    write_json(previous / 'repair-jobs.json', [pilot])
    write_json(previous / 'result.json', {'exit_code': 0, 'original_slots': slots})
    def price(cell, source_identity):
        assert source_identity == 'source'
        final, full = states.get(str(cell), (False, False))
        return {'cell': str(cell), 'design': read_json(cell / 'isolated_design.json'),
                'final_assessment': final, 'status': 'complete_workload' if full else 'incomplete',
                'seconds': 9000., 'posterior_outcome_used_for_pricing': False}
    monkeypatch.setattr(campaign, 'fit_price', price)
    return campaign.ContinuationInventory(root, previous, 'source'), states, assessment, pilot


def test_stage_carryover_unlocks_matching_main_and_preserves_negative_completion(continuation_inventory):
    inv, states, assessment, pilot = continuation_inventory
    initial = file_hash(assessment)
    jobs = inv.refresh([])
    assert len(inv.snapshot['original_slots']) == 32
    assert any(j['case'] == 'K0' and j['design'] is None for j in jobs)
    assert any(j['case'] == 'K2' and j['design'] is None for j in jobs)
    assert not any(j['cell'] == str(assessment.parents[1]) for j in jobs)
    assert [j['job_id'] for j in jobs if j['case'] == 'K1'] == ['pilot-K1']
    assert all(j['first_quantum_seconds'] <= campaign.INITIAL_QUANTUM for j in jobs)
    rows = [{'job_id': j['job_id'], 'status': 'timed_out', 'attempts': []} for j in jobs]
    states[pilot['cell']] = (True, True)
    rows[0]['status'] = 'complete'
    new = inv.refresh(rows)
    assert len(new) == 4 and all(j['case'] == 'K1' for j in new)
    assert not inv.refresh(rows + [{'job_id': j['job_id'], 'status': 'failed'} for j in new])
    assert file_hash(assessment) == initial


def test_negative_incomplete_pilot_workload_does_not_unlock_main(continuation_inventory):
    inv, states, _, pilot = continuation_inventory
    states[pilot['cell']] = (True, False)
    assert not any(j['case'] == 'K1' for j in inv.refresh([]))
    assert all(s['status'] == 'unpriced_workload' for s in inv.snapshot['original_slots'] if s['case'] == 'K1')


def test_completed_assessment_mutation_vetoes_continuation(continuation_inventory):
    inv, _, assessment, _ = continuation_inventory
    assessment.write_text('{}')
    with pytest.raises(ValueError, match='preserved final assessment'):
        inv.refresh([])


def test_reference_resolution_and_repaired_preparation_must_match(continuation_inventory):
    inv, states, _, _ = continuation_inventory
    path = campaign.PREPARED / 'pricing/price-K3/isolated_design.json'
    value = read_json(path)
    value['options']['reference_settings'] = {'resolution': 321}
    write_json(path, value)
    jobs = inv.refresh([])
    assert not any(j['case'] in {'K3', 'K5', 'K6'} for j in jobs)
    for case in ['K5', 'K6']:
        path = campaign.PREPARED / 'pricing' / ('price-' + case) / 'isolated_design.json'
        write_json(path, inv.main_design(read_json(path)))
    jobs = inv.refresh([])
    repaired = [j for j in jobs if j['case'] in {'K5', 'K6'}]
    assert len(repaired) == 8
    assert all(j['design']['options']['preparation_max_restarts'] == 3 for j in repaired)


@pytest.mark.parametrize('fail_at', [None, 'manifest', 'pool'])
def test_worker_settles_once_after_success_or_exception(tmp_path, monkeypatch, fail_at):
    ledger, root = tmp_path / 'ledger.json', tmp_path / 'run'
    write_json(ledger, {'grant_gpu_seconds': 30000., 'remaining_gpu_seconds': 29000.,
        'records': [{'receipt': 'prior', 'elapsed_seconds': 1000.}]})
    monkeypatch.setenv('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-fixture')
    now = [100.]
    monkeypatch.setattr(campaign, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    def manifest(*args):
        if fail_at == 'manifest':
            now[0] += 2.
            raise ValueError('broken manifest')
        return {}
    monkeypatch.setattr(campaign, 'coordinator_manifest', manifest)
    monkeypatch.setattr(campaign, 'check_inputs', lambda s: {'identity': 'source'})
    class Inventory:
        pilots, preserved_assessments = [], {}
        snapshot = {'original_slots': [], 'pilots': []}
        def __init__(self, *args): pass
        def refresh(self, rows): return []
        def check_devices(self): pass
    monkeypatch.setattr(campaign, 'ContinuationInventory', Inventory)
    def pool(*args, **kwargs):
        assert kwargs['max_attempts'] is None
        assert read_json(ledger)['active_reservation']['service'] == 'test-progress'
        now[0] += 60.
        if fail_at == 'pool': raise ValueError('broken checkpoint')
        return {'rows': [], 'queue_closed': True}
    monkeypatch.setattr(campaign, 'run_pool', pool)
    args = SimpleNamespace(output=root, ledger=ledger, previous_run=tmp_path / 'previous',
        gpu='GPU-fixture', service='test-progress', seconds=12000., deadline=12100.)
    if fail_at:
        with pytest.raises(ValueError, match='broken'): campaign.worker(args)
    else:
        campaign.worker(args)
    settled = read_json(ledger)
    assert 'active_reservation' not in settled
    assert len(settled['records']) == 2
    assert settled['charged_gpu_seconds'] == 1000. + campaign.CLOSEOUT + (2. if fail_at == 'manifest' else 60.)
    assert read_json(root / 'result.json')['exit_code'] == (1 if fail_at else 0)
    campaign.settle(ledger, read_json(root / 'execution.json'))
    assert read_json(ledger) == settled


def test_grant_carryover_is_checked_before_allocation():
    ledger = {'grant_gpu_seconds': 532800., 'remaining_gpu_seconds': 244012.31672476209,
              'records': [{'receipt': 'old', 'elapsed_seconds': 288787.6832752379}]}
    assert campaign.available_budget(ledger) == campaign.GRANT_CAP
    ledger['records'].append({'receipt': 'new-attempt', 'elapsed_seconds': 1200.,
                             'campaign_id': campaign.CAMPAIGN_ID})
    ledger['remaining_gpu_seconds'] -= 1200.
    assert campaign.available_budget(ledger) == campaign.GRANT_CAP - 1200.
    ledger['records'].append(ledger['records'][0])
    with pytest.raises(ValueError, match='duplicate'): campaign.available_budget(ledger)


def test_master_progress_preserves_other_work_and_clears_terminal_service(tmp_path):
    progress, ledger = tmp_path / 'progress.json', tmp_path / 'ledger.json'
    write_json(progress, {'unrelated': {'preserve': True}, 'remaining_budget_seconds': {'cpu_reference': 10.}})
    write_json(ledger, {'remaining_gpu_seconds': 300., 'active_reservation': {'service': 'running'}})
    args = SimpleNamespace(progress_file=progress, ledger=ledger, output=tmp_path / 'run', service='running')
    slots = [{'status': 'preserved_complete'}, {'status': 'queued'}]
    campaign.publish_master(args, 'running', slots, 'next-fit')
    d = read_json(progress)
    assert d['unrelated'] == {'preserve': True}
    assert d['original_main_complete_assessments'] == 1
    assert d['active_service'] == 'running' and d['current_job'] == 'next-fit'
    write_json(ledger, {'remaining_gpu_seconds': 200.})
    campaign.publish_master(args, 'closed', slots)
    d = read_json(progress)
    assert d['active_service'] is None and d['active_gpu_reservation'] is None
    assert d['remaining_budget_seconds'] == {'cpu_reference': 10., 'gpu': 200.}


def test_wrong_gpu_checkpoint_is_rejected_before_dispatch(tmp_path, monkeypatch):
    cell = tmp_path / 'fit'
    write_json(cell / 'replication-0000/tuning/execution_spec.json',
               {'execution': {'runtime_policy': {'device_type': 'GPU', 'cuda_visible_devices': 'GPU-original'}}})
    monkeypatch.setenv('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-other')
    with pytest.raises(ValueError, match='original GPU'):
        recovery.check_checkpoint_device(cell)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-original')
    recovery.check_checkpoint_device(cell)
    monkeypatch.setenv('TF_FORCE_GPU_ALLOW_GROWTH', 'false')
    with pytest.raises(ValueError, match='memory growth'):
        recovery.check_checkpoint_device(cell)


@pytest.mark.parametrize('mutation', [None, 'checkpoint', 'numerical_evidence', 'progress', 'different_failure'])
def test_diagnosed_device_startup_retry_preserves_all_costs_and_evidence(tmp_path, design, monkeypatch, mutation):
    source, cell, d, identity = saved_fit(tmp_path, design)
    path = cell / 'replication-0000'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-original')
    monkeypatch.setenv('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
    write_json(path / 'tuning/execution_spec.json',
               {'execution': {'runtime_policy': {'device_type': 'GPU', 'cuda_visible_devices': 'GPU-original'}}})
    attempt = tmp_path / 'failed-attempt'
    write_json(attempt / 'before/tuning/tuning_checkpoint.json', read_json(path / 'tuning/tuning_checkpoint.json'))
    write_json(attempt / 'preserved-evidence.json', recovery.immutable_fit_evidence(path))
    prefix = path / 'process-attempt-002'
    failed = {'status': 'failed', 'elapsed_seconds': 2., 'exit_code': 1, 'campaign_continuation': True,
        'preserved_evidence_verified': True, 'progress': {}, 'allocation_file': str(attempt / 'allocation.json')}
    write_json(str(prefix) + '-launch.json', {'command': 'fixture'})
    write_json(str(prefix) + '-manifest.json', {'design_identity': d.identity,
        'source': {'identity': identity['source_identity']}, 'runtime': {}})
    write_json(str(prefix) + '-failure.json', {'exception': 'ValueError',
        'reason': 'execution device or numerical policy mismatch' if mutation != 'different_failure' else 'bad state'})
    if mutation == 'progress': failed['progress'] = stopped()['progress']
    write_json(str(prefix) + '-exit.json', failed)
    receipt_hash = file_hash(str(prefix) + '-exit.json')
    if mutation == 'checkpoint': write_json(path / 'tuning/tuning_checkpoint.json', {'changed': True})
    if mutation == 'numerical_evidence': write_json(path / 'members/unfavorable/result.json', {'posterior_passed': True})
    calls = []
    def supervise(command, log, seconds, device, **kwargs):
        assert mutation is None
        calls.append(command)
        p = str(log).removesuffix('.log')
        write_json(p + '-manifest.json', {'design_identity': d.identity,
            'source': {'identity': identity['source_identity']}, 'runtime': {}})
        write_json(path / 'independent_assessment.json', {'posterior_passed': False})
        return {'status': 'complete', 'elapsed_seconds': 1., 'exit_code': 0}
    monkeypatch.setattr(recovery, 'supervise_fit', supervise)
    args = dict(cell=cell, source=source, replication=0, output=tmp_path / 'retry',
        cumulative_cap_seconds=18., quantum_seconds=4., deadline=time.monotonic() + 10.,
        max_additional_attempts=2, repair_device_startup_failure=True)
    if mutation in {'checkpoint', 'numerical_evidence', 'progress'}:
        with pytest.raises(ValueError): recovery.continue_frozen_fit(**args)
    else:
        result = recovery.continue_frozen_fit(**args)
        if mutation:
            assert result['status'] == 'ineligible_or_allocation_exhausted'
        else:
            assert result['status'] == 'complete' and result['cumulative_fit_seconds'] == 13.
            allocation = read_json(tmp_path / 'retry/allocation.json')
            assert allocation['diagnosed_startup_repair']['prior_costs_retained']
    assert len(calls) == (1 if mutation is None else 0)
    assert file_hash(str(prefix) + '-exit.json') == receipt_hash
