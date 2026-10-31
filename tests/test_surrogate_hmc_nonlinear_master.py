"""M10 orchestration, evidence reuse and cumulative preparation accounting."""
import json
from pathlib import Path

import pytest

from bayesfilter.inference import surrogate_hmc_program as program


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def attempt(root, label, cpu, wall, status='EXECUTION_FAILED'):
    output = root/'docs/plans/artifacts'/('surrogate-hmc-all-models-test-m10-reference-'+label)
    save(output/'accounting.json', dict(charged_this_run=dict(gpu_seconds=0., cpu_worker_seconds=cpu)))
    save(output/'run-manifest.json', dict(wall_seconds=wall,
        command=['python', 'scripts/prepare_surrogate_hmc_nonlinear_reference.py']))
    save(output/'report.json', dict(status=status))
    return output


def test_m10_retry_charges_standalone_master_and_test_costs_once(tmp_path):
    attempt(tmp_path, '01', 10., 20.)
    attempt(tmp_path, 'tests-01', 3., 4.)
    nested = tmp_path/'docs/plans/artifacts/surrogate-hmc-all-models-test-all/P3-M10-reference'
    save(nested/'accounting.json', dict(charged_this_run=dict(cpu_worker_seconds=8., gpu_seconds=0.)))
    save(nested.parent/'accounting.json', dict(jobs=[dict(
        command=['python', 'scripts/prepare_surrogate_hmc_nonlinear_reference.py', '--output', str(nested)],
        cpu_worker_seconds=8., wall_seconds=9.)]))
    state = program.m10_reference_state(tmp_path)
    assert state['attempts'] == 2
    assert state['family_cpu_remaining'] == program.M10_CPU_CAP-21.
    assert state['cpu_worker_seconds'] == state['wall_seconds'] == 0.


def test_m10_scientific_failure_is_not_an_infrastructure_retry(tmp_path):
    attempt(tmp_path, '01', 3., 4., 'REFERENCE_CHECK_FAILED')
    with pytest.raises(ValueError, match='prospective repair'):
        program.m10_reference_state(tmp_path)


def test_reusing_preparation_preserves_dependency_execution(tmp_path, monkeypatch):
    report = tmp_path/'reference.json'
    save(report, dict(status='QUALIFIED_LOCAL_LIKELIHOOD_REFERENCE'))
    prior = dict(report=str(report), report_sha256=program.file_hash(report))
    launched = []
    def run(command, *args, **kwargs):
        launched.append(command[0])
        return dict(status='COMPLETED')
    monkeypatch.setattr(program, 'run_job', run)
    jobs = [dict(name='reference', prior_preparation_result=prior),
            dict(name='consumer', depends_on=['reference'], command=['consumer'], device='cpu', timeout=1.)]
    result = program.execute_job_graph(jobs, tmp_path, {})
    assert launched == ['consumer']
    assert result['reference']['reused_preparation']
    assert result['reference']['authorizes_new_inference'] is False
    report.write_text('{}')
    result = program.execute_job_graph(jobs, tmp_path, {})
    assert result['reference']['status'] == 'BLOCKED_PREFLIGHT'
    assert result['consumer']['status'] == 'SKIPPED_PREREQUISITE'


def test_m10_master_selects_cpu_stage_without_posterior_promotion(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(program, 'm10_posterior_state', lambda root: dict(qualified=None))
    monkeypatch.setattr(program, 'm10_consumer_state', lambda root: dict(qualified=None))
    monkeypatch.setattr(program, 'm10_reference_state', lambda root: dict(
        wall_seconds=300., cpu_worker_seconds=300., qualified=None))
    assert program.main(['--stage', 'P3-M10-reference', '--scope', 'M10',
                         '--output', str(tmp_path/'unused'), '--dry-run']) == 0
    launch = json.loads(capsys.readouterr().out)
    assert len(launch['jobs']) == 1
    assert launch['jobs'][0]['device'] == 'cpu'
    assert Path(launch['jobs'][0]['command'][1]).name == 'prepare_surrogate_hmc_nonlinear_reference.py'
    card = launch['scope_cards']['M10']
    assert card['posterior_reference_qualified'] is False
    assert card['surrogate_hmc_tested'] is False
    assert card['chart'] == 'identity log-noise scales'


def test_consumer_accounting_combines_standalone_and_mixed_master_once(tmp_path):
    output = tmp_path/'docs/plans/artifacts/surrogate-hmc-all-models-test-m10-consumer-01'
    save(output/'accounting.json', dict(charged_this_run=dict(cpu_worker_seconds=10., gpu_seconds=20.)))
    save(output/'run-manifest.json', dict(command=['python', 'qualify_surrogate_hmc_nonlinear_consumer.py']))
    save(output/'report.json', dict(status='EXECUTION_FAILED'))
    nested = output.parent/'surrogate-hmc-all-models-mixed/P3-M10-qualify'
    save(nested/'accounting.json', dict(charged_this_run=dict(cpu_worker_seconds=8., gpu_seconds=9.)))
    save(nested.parent/'accounting.json', dict(jobs=[dict(
        command=['python', 'qualify_surrogate_hmc_nonlinear_consumer.py', '--output', str(nested)],
        cpu_worker_seconds=8., gpu_seconds=9.)]))
    state = program.m10_consumer_state(tmp_path)
    assert state['attempts'] == 2
    assert state['wall_seconds'] == state['cpu_worker_seconds'] == 0.
    assert state['qualified'] is None


def test_consumer_reuse_rejects_missing_check_and_source_drift(tmp_path):
    output = tmp_path/'docs/plans/artifacts/surrogate-hmc-all-models-test-m10-consumer-01'
    sources = {}
    for name in ('bayesfilter/testing/surrogate_hmc_nonlinear_consumer.py',
                 'bayesfilter/inference/neural_force_hmc.py'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# source\n')
        sources[name] = program.file_hash(path)
    save(output/'accounting.json', dict(charged_this_run=dict(cpu_worker_seconds=10., gpu_seconds=20.)))
    save(output/'data.json', {'observations': [0., 1.]})
    save(output/'reference/run-manifest.json', {'role': 'local reference'})
    save(output/'run-manifest.json', dict(command=['python', 'qualify_surrogate_hmc_nonlinear_consumer.py'],
        source_hashes=sources, data_file=str(output/'data.json'), data_sha256=program.file_hash(output/'data.json'),
        parent_reference=str(output/'reference'), parent_manifest_sha256=program.file_hash(output/'reference/run-manifest.json')))
    checks = {name: True for name in program.M10_CONSUMER_CHECKS}
    report = dict(status='QUALIFIED_CONSUMER_MECHANICS', qualified=True, checks=checks)
    save(output/'report.json', report)
    state = program.m10_consumer_state(tmp_path)
    assert state['wall_seconds'] == 580.
    assert state['cpu_worker_seconds'] == 440.
    assert state['qualified']['authorizes_new_inference'] is False
    del checks['row_independence']
    save(output/'report.json', report)
    with pytest.raises(ValueError, match='incomplete M10 mechanics'):
        program.m10_consumer_state(tmp_path)
    checks['row_independence'] = True
    save(output/'report.json', report)
    (tmp_path/'bayesfilter/inference/neural_force_hmc.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='source changed'):
        program.m10_consumer_state(tmp_path)
    repair_state = program.m10_consumer_state(tmp_path, for_requalification=True)
    assert repair_state['qualified'] is None
    assert repair_state['attempts'] == 1
    assert repair_state['superseded_mechanics'] == [str((output/'report.json').resolve())]
    # Only a fresh current-source qualification can supersede old mechanics.
    fresh = output.with_name(output.name[:-2]+'02')
    manifest = json.loads((output/'run-manifest.json').read_text())
    manifest['source_hashes'] = {name: program.file_hash(tmp_path/name) for name in sources}
    save(fresh/'run-manifest.json', manifest)
    save(fresh/'report.json', report)
    save(fresh/'accounting.json', dict(charged_this_run=dict(cpu_worker_seconds=5., gpu_seconds=8.)))
    repaired = program.m10_consumer_state(tmp_path)
    assert repaired['qualified']['report'] == str((fresh/'report.json').resolve())
    assert repaired['attempts'] == 2
    assert repaired['family_remaining']['cpu_worker_seconds'] == program.M10_CPU_CAP-15.


def test_master_reuses_mechanics_without_sampling(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(program, 'm10_posterior_state', lambda root: dict(qualified=None))
    monkeypatch.setattr(program, 'm10_reference_state', lambda root: dict(
        wall_seconds=300., cpu_worker_seconds=300., qualified={'report': 'reference'}))
    monkeypatch.setattr(program, 'm10_consumer_state', lambda root: dict(
        wall_seconds=500., cpu_worker_seconds=400., qualified={'report': 'mechanics', 'authorizes_new_inference': False}))
    assert program.main(['--stage', 'P3-M10-qualify', '--scope', 'M10',
                         '--output', str(tmp_path/'unused'), '--dry-run']) == 0
    launch = json.loads(capsys.readouterr().out)
    job, = launch['jobs']
    assert job['prior_preparation_result']['report'] == 'mechanics'
    assert job['prior_preparation_result']['authorizes_new_inference'] is False
    assert launch['scope_cards']['M10']['surrogate_hmc_tested'] is False


def test_posterior_budget_counts_mixed_children_once_and_tests_only_in_family(tmp_path):
    base=tmp_path/'docs/plans/artifacts'
    standalone=base/'surrogate-hmc-all-models-test-m10-posterior-01'
    script='prepare_surrogate_hmc_nonlinear_posterior.py'
    save(standalone/'accounting.json',dict(charged_this_run=dict(gpu_seconds=5.,cpu_worker_seconds=11.)))
    save(standalone/'run-manifest.json',dict(command=['python',script],wall_seconds=6.))
    save(standalone/'report.json',dict(status='GRID_NUMERICAL_FAILURE'))
    save(base/'surrogate-hmc-all-models-test-m10-posterior-tests-01/accounting.json',
         dict(charged_this_run=dict(gpu_seconds=0.,cpu_worker_seconds=7.)))
    mixed=base/'surrogate-hmc-all-models-test-mixed'
    nested=mixed/'P3-M10-posterior'
    save(nested/'accounting.json',dict(charged_this_run=dict(gpu_seconds=50.,cpu_worker_seconds=50.)))
    save(nested/'report.json',dict(status='EXECUTION_FAILED'))
    save(mixed/'accounting.json',dict(charged_this_run=dict(gpu_seconds=4.,cpu_worker_seconds=4.),jobs=[
        dict(command=['python',script,'--output',str(nested)],gpu_seconds=4.,cpu_worker_seconds=3.,wall_seconds=4.)]))
    state=program.m10_posterior_state(tmp_path)
    assert state['attempts']==2 and state['qualified'] is None
    assert state['cpu_worker_seconds']==836. and state['wall_seconds']==790.
    assert program.m10_consumer_state(tmp_path)['family_remaining']['cpu_worker_seconds']==program.M10_CPU_CAP-21.


def test_posterior_reuse_checks_precision_source_data_and_grids(tmp_path):
    output=tmp_path/'docs/plans/artifacts/surrogate-hmc-all-models-test-m10-posterior-01'
    names=('bayesfilter/testing/surrogate_hmc_nonlinear_posterior_reference.py',
           'bayesfilter/testing/surrogate_hmc_nonlinear_quadrature.py',
           'bayesfilter/testing/surrogate_hmc_nonlinear_reference.py',
           'bayesfilter/score_study/nonlinear_tf.py','bayesfilter/score_study/gaussian_tf.py')
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original')
    data=output/'data.json';save(data,dict(role='claim',regime='nominal'))
    grids=[]
    for n,s,g in ((40,20,128),(64,20,128),(96,20,128),(96,32,128),(96,32,64)):
        path=output/f'grid-{n}-{s}-{g}.json';save(path,dict(saved=True))
        grids.append(dict(order=n,state_order=s,normal_order=g,path=str(path),sha256=program.file_hash(path)))
    save(output/'accounting.json',dict(charged_this_run=dict(gpu_seconds=10.,cpu_worker_seconds=12.)))
    save(output/'run-manifest.json',dict(command=['python','prepare_surrogate_hmc_nonlinear_posterior.py'],
         wall_seconds=10.,data_file=str(data),data_sha256=program.file_hash(data),
         source_hashes={name:program.file_hash(tmp_path/name) for name in names}))
    report=dict(status='QUALIFIED_MODEL_POSTERIOR_REFERENCE',qualified=True,model='M10',partition='claim-nominal',
                checks={k:True for k in program.M10_POSTERIOR_CHECKS},margins=[.1]*5+[.02]*6+[.03]*2,
                function_allowance=[.0001]*13,allowance_is_rigorous_bound=False,grids=grids)
    save(output/'report.json',report)
    qualified=program.m10_posterior_state(tmp_path)['qualified']
    assert qualified['posterior_reference_qualified'] is True
    assert qualified['authorizes_new_inference'] is False
    report['function_allowance'][5]=.003;save(output/'report.json',report)
    with pytest.raises(ValueError,match='precision'):program.m10_posterior_state(tmp_path)
    report['function_allowance'][5]=.0001;save(output/'report.json',report)
    for path,match in ((tmp_path/names[0],'source changed'),(data,'data changed'),(Path(grids[0]['path']),'grid changed')):
        original=path.read_text();path.write_text('changed')
        with pytest.raises(ValueError,match=match):program.m10_posterior_state(tmp_path)
        path.write_text(original)
    del report['checks']['function_precision'];save(output/'report.json',report)
    with pytest.raises(ValueError,match='incomplete M10 posterior'):program.m10_posterior_state(tmp_path)


def test_master_reuses_posterior_without_hmc_authority(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(program,'m10_reference_state',lambda root:dict(qualified={'report':'reference'}))
    monkeypatch.setattr(program,'m10_consumer_state',lambda root:dict(qualified=None))
    monkeypatch.setattr(program,'m10_posterior_state',lambda root:dict(wall_seconds=0.,cpu_worker_seconds=0.,
        qualified=dict(report='posterior',posterior_reference_qualified=True,authorizes_new_inference=False)))
    assert program.main(['--stage','P3-M10-posterior','--scope','M10','--output',str(tmp_path/'unused'),'--dry-run'])==0
    launch=json.loads(capsys.readouterr().out);job,=launch['jobs']
    assert Path(job['command'][1]).name=='prepare_surrogate_hmc_nonlinear_posterior.py'
    assert job['device']=='gpu' and '--gpu' in job['command']
    assert job['prior_preparation_result']['authorizes_new_inference'] is False
    assert launch['scope_cards']['M10']['posterior_reference_qualified'] is True
    assert launch['scope_cards']['M10']['surrogate_hmc_tested'] is False


def test_m10_qualified_history_checks_numerical_source_and_data(tmp_path):
    output = attempt(tmp_path, '01', 10., 20.)
    hashes = {}
    for name in ('bayesfilter/testing/surrogate_hmc_nonlinear_reference.py',
                 'bayesfilter/score_study/nonlinear_tf.py', 'bayesfilter/score_study/gaussian_tf.py'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# source\n')
        hashes[name] = program.file_hash(path)
    save(output/'run-manifest.json', dict(wall_seconds=20., source_hashes=hashes,
        command=['python', 'scripts/prepare_surrogate_hmc_nonlinear_reference.py']))
    save(output/'data.json', {'observations': [0., 1.]})
    save(output/'grid.json', {'value': 1.})
    checks = {name: True for name in ('mesh_refinement', 'domain_refinement', 'domain_loss', 'posterior_edge',
        'linear_identity', 'analytical_noise_scores', 'independent_nonlinear_refinement', 'independent_nonlinear_agreement')}
    save(output/'report.json', dict(status='QUALIFIED_LOCAL_LIKELIHOOD_REFERENCE', checks=checks,
        partitions={'example': dict(data_file=str(output/'data.json'), data_sha256=program.file_hash(output/'data.json'))},
        grids=[dict(values_file='grid.json', sha256=program.file_hash(output/'grid.json'))]))
    state = program.m10_reference_state(tmp_path)
    assert state['qualified']['posterior_reference_qualified'] is False
    (output/'data.json').write_text('{}')
    with pytest.raises(ValueError, match='data changed'):
        program.m10_reference_state(tmp_path)
