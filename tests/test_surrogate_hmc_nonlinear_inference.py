"""M10 posterior reporting rejects mismatched evidence and closed-look repeats."""
import json
from pathlib import Path

import pytest

from bayesfilter.inference import surrogate_hmc_program as program


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record(directory, name):
    path = directory/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
    return dict(path=name, sha256=program.file_hash(path))


def preserved(root, nested=False):
    output = root/'docs/plans/artifacts/surrogate-hmc-all-models-test-m10-inference-01'
    if nested:
        output = output/'P3-M10-inference'
    source = record(output/'source', 'runner.py')
    inputs = {}
    for name in ('claim', 'calibration', 'selection', 'posterior_report', 'reference_report', 'endpoint_report'):
        path = output/'inputs'/f'{name}.json'
        save(path, dict(role=name))
        inputs[name] = dict(path=str(path), sha256=program.file_hash(path))
    save(output/'data.json', dict(role='claim'))
    save(output/'run-manifest.json', dict(scope='M10-T2-N1024', exit_code=0,
        command=['python', 'scripts/run_surrogate_hmc_nonlinear.py'],
        source_hashes={'runner.py': source['sha256']}, source_drift=[], inputs=inputs,
        data_sha256=program.file_hash(output/'data.json'), retained_looks=[1000, 2000],
        function_count=13, comparison_families=4))
    save(output/'accounting.json', dict(exit_code=0))
    binding = dict(binding_hash='binding', target_scope='M10-T2-N1024-claim-attempt1:B')
    save(output/'binding.json', binding)
    tune_path = output/'B/tuning/manifest.json'
    save(tune_path.parent/'nomination/manifest.json', dict(passed=True))
    trace = record(tune_path.parent, 'states.tfbin')
    save(tune_path, dict(schema='bayesfilter.verified_tensorflow_hmc.v1', artifact_authority=True,
        handoff_eligible=True, passed=True, checks=dict(fresh_rhat=True), runner_binding=binding,
        nomination_manifest_sha256=program.file_hash(tune_path.parent/'nomination/manifest.json'),
        warmup=dict(states=trace), verification=dict(states=trace)))
    chunks = []
    for index in range(8):
        directory = output/'B'/f'chunk-{index+1:02d}'
        sample = record(directory, 'samples.tfbin')
        trace = record(directory, 'states.tfbin')
        initial = record(directory, 'initial.tfbin')
        previous = chunks[-1]['manifest'] if chunks else None
        save(directory/'manifest.json', dict(binding_hash='binding', health_passed=True,
            checks=dict(finite=True), tuning_manifest_sha256=program.file_hash(tune_path),
            continuation_manifest=previous, config=dict(continuation_manifest=previous,
            num_results=250, seed=[20260922, 33000+index]), raw_samples=sample,
            initial_state=initial, chunk=dict(states=trace)))
        chunks.append(dict(manifest=str(directory/'manifest.json'), health_passed=True))
    row = dict(tuning=dict(passed=True, manifest=str(tune_path)), chunks=chunks, retained_draws=2000,
        looks=[dict(draws_per_chain=n, status='INCONCLUSIVE', diagnostic_valid=False) for n in (1000, 2000)],
        status='INCONCLUSIVE')
    save(output/'report.json', dict(status='COMPLETED_BOUNDED_INFERENCE', model_id='M10-T2', arms=dict(B=row),
        comparisons={'B/R': 'INCONCLUSIVE', 'C/D': 'NOT_RUN', 'D/R': 'NOT_RUN', 'C/R': 'NOT_RUN'}))
    return output


@pytest.mark.parametrize('nested', [False, True])
def test_preserved_attempt_is_reported_without_new_inference_authority(tmp_path, nested):
    output = preserved(tmp_path, nested)
    # An unrelated current-source edit must not erase archived evidence.
    (tmp_path/'runner.py').write_text('new reporting source')
    state = program.m10_inference_history(tmp_path)
    assert state['artifact_valid'] is True
    assert state['comparisons']['B/R'] == 'INCONCLUSIVE'
    assert state['arms']['B']['retained_draws'] == 2000
    assert state['authorizes_new_inference'] is False
    assert state['report'] == str((output/'report.json').resolve())


@pytest.mark.parametrize('corruption', ['source', 'data', 'input', 'sample', 'trace', 'tuning_trace',
    'binding', 'continuation', 'seed', 'repeat_look', 'continued_closed', 'invalid_equivalence', 'other_comparison'])
def test_invalid_evidence_cannot_be_reported_as_a_completed_test(tmp_path, corruption):
    output = preserved(tmp_path)
    files = {'source': output/'source/runner.py', 'data': output/'data.json',
        'input': output/'inputs/claim.json', 'sample': output/'B/chunk-01/samples.tfbin',
        'trace': output/'B/chunk-01/states.tfbin', 'tuning_trace': output/'B/tuning/states.tfbin'}
    if corruption in files:
        files[corruption].write_text('corrupted')
    elif corruption in ('binding', 'continuation', 'seed'):
        path = output/'B/chunk-02/manifest.json'
        v = json.loads(path.read_text())
        if corruption == 'seed':
            v['config']['seed'] = [20260922, 33000]
        else:
            v['binding_hash' if corruption == 'binding' else 'continuation_manifest'] = 'wrong'
        save(path, v)
    else:
        path = output/'report.json'
        v = json.loads(path.read_text())
        row = v['arms']['B']
        if corruption == 'repeat_look':
            row['looks'][1]['draws_per_chain'] = 1000
        elif corruption == 'continued_closed':
            row['looks'][0].update(status='EQUIVALENT', diagnostic_valid=True)
        elif corruption == 'invalid_equivalence':
            row['looks'][1]['status'] = 'EQUIVALENT'
            row['status'] = v['comparisons']['B/R'] = 'EQUIVALENT'
        else:
            v['comparisons']['C/D'] = 'EQUIVALENT'
        save(path, v)
    assert program.m10_inference_history(tmp_path)['status'] == 'ARTIFACT_INVALID'


def test_failed_or_in_progress_attempt_is_preserved(tmp_path):
    output = preserved(tmp_path)
    path = output/'run-manifest.json'
    v = json.loads(path.read_text())
    del v['exit_code']
    save(path, v)
    assert program.m10_inference_history(tmp_path)['status'] == 'RUNNING'
    v['exit_code'] = 1
    save(path, v)
    save(output/'report.json', dict(status='EXECUTION_FAILED', model_id='M10-T2'))
    state = program.m10_inference_history(tmp_path)
    assert state['status'] == 'EXECUTION_FAILED'
    assert state['authorizes_new_inference'] is False


def test_two_attempts_require_reconciliation(tmp_path):
    preserved(tmp_path)
    preserved(tmp_path, nested=True)
    assert program.m10_inference_history(tmp_path)['status'] == 'RECONCILIATION_REQUIRED'


def test_master_launch_manifest_does_not_count_as_an_inference_attempt(tmp_path):
    output = tmp_path/'docs/plans/artifacts/surrogate-hmc-all-models-test-m10-inference-01'
    save(output/'run-manifest.json', dict(schema=program.SCHEMA, scope='M10', jobs=[],
        command_line=['python', 'scripts/run_surrogate_hmc_program.py']))
    assert program.m10_inference_history(tmp_path)['status'] == 'NOT_RUN'
    preserved(tmp_path, nested=True)
    state = program.m10_inference_history(tmp_path)
    assert state['artifact_valid'] is True
    assert len(state['attempt_manifests']) == 1


def test_master_dispatches_inference_and_checks_whole_schedule(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(program, 'ROOT', tmp_path)
    monkeypatch.setattr(program, 'm10_reference_state', lambda _: dict(qualified=None))
    monkeypatch.setattr(program, 'm10_controls_state', lambda _: dict(qualified=None))
    monkeypatch.setattr(program, 'm10_posterior_state', lambda _: dict(qualified=None))
    monkeypatch.setattr(program, 'm10_consumer_state', lambda _: dict(qualified=None,
        family_remaining=dict(cpu_worker_seconds=3300., gpu_seconds=6400.)))
    forecast = dict(required=dict(cpu_worker_seconds=3006., wall_seconds=3028.))
    monkeypatch.setattr(program, 'm10_endpoint_state', lambda _: dict(qualified=dict(complete_forecast=forecast)))
    args = ['--stage', 'P3-M10-inference', '--scope', 'M10', '--output', str(tmp_path/'out'), '--dry-run']
    assert program.main(args) == 0
    launch = json.loads(capsys.readouterr().out)
    job, = launch['jobs']
    assert job['cpu_allowance'] == 3200.
    assert Path(job['command'][1]).name == 'run_surrogate_hmc_nonlinear.py'
    assert launch['inference_forecast'] == forecast
    forecast['required']['cpu_worker_seconds'] = 3201.
    with pytest.raises(SystemExit):
        program.main(args)


def test_public_verified_tuning_budget_and_reference_functional_units():
    import tensorflow as tf
    from scripts.run_surrogate_hmc_nonlinear import tuning_config
    from bayesfilter.inference import HMC_TUNING_INTERFACE_CAPABILITIES
    from bayesfilter.testing.surrogate_hmc_nonlinear_inference_metrics import scalar_functionals
    config = tuning_config('M10-test')
    payload = config.payload()
    assert payload['warmup_results'] == 250
    assert payload['verification_results'] == 1000
    assert tuple(payload['nomination']['trajectory_candidates']) == (1, 2)
    assert any(c.interface_name == 'tune_hmc_kernel' and c.artifact_authority for c in HMC_TUNING_INTERFACE_CAPABILITIES)
    tensor = lambda value: tf.constant(value, tf.float64)
    actual = scalar_functionals(tensor([[1, 2], [3, -2]]), tensor([[.2, .3], [.4, .5]]),
        tensor([1, 2]), tensor([2, 4]), tensor([[0, 1, 2], [0, 2, 4]]))
    expected = tensor([[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, .2, .3],
        [1, -1, 1, 1, -1, 0, 0, 0, 1, 1, 1, .4, .5]])
    tf.debugging.assert_near(actual, expected, atol=1e-14)
