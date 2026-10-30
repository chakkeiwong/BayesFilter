"""SV20 posterior reporting rejects mismatched evidence and closed-look repeats."""
import json
from pathlib import Path

import pytest

from bayesfilter.inference import surrogate_hmc_sv_program as program


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record(directory, name):
    path = directory/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
    return dict(path=name, sha256=program.file_hash(path))


def preserved(root, nested=False):
    output = root/'docs/plans/artifacts/surrogate-hmc-all-models-test-sv20-inference-01'
    if nested:
        output = output/'P3-SV20-inference'
    source = record(output/'source', 'runner.py')
    inputs = {}
    for name in sorted(program.INPUT_NAMES | {'endpoint_report', 'endpoint_manifest'}):
        path = output/'inputs'/f'{name}.json'
        save(path, dict(role=name))
        inputs[name] = dict(path=str(path), sha256=program.file_hash(path))
    save(output/'data.json', dict(role='claim'))
    save(output/'run-manifest.json', dict(scope='SV20-T20-N1024', exit_code=0,
        command=['python', 'scripts/run_surrogate_hmc_sv.py'],
        source_hashes={'runner.py': source['sha256']}, source_drift=[], inputs=inputs,
        data_sha256=program.file_hash(output/'data.json'), retained_looks=[1000, 2000],
        function_count=13, comparison_families=4))
    save(output/'accounting.json', dict(exit_code=0))
    binding = dict(binding_hash='binding', target_scope='SV20-T20-N1024-claim-attempt1:B')
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
        force = record(directory, 'force_fallback.tfbin')
        initial = record(directory, 'initial.tfbin')
        previous = chunks[-1]['manifest'] if chunks else None
        save(directory/'manifest.json', dict(binding_hash='binding', health_passed=True,
            checks=dict(finite=True), tuning_manifest_sha256=program.file_hash(tune_path),
            continuation_manifest=previous, config=dict(continuation_manifest=previous,
            num_results=250, seed=[20260922, 36000+index]), raw_samples=sample,
            initial_state=initial, chunk=dict(states=trace, force_fallback=force)))
        chunks.append(dict(manifest=str(directory/'manifest.json'), health_passed=True,
            original_force_health=dict(passed=True, force_fallback_count_by_chain=[0]*4,
                manifest_sha256=program.file_hash(directory/'manifest.json'),
                telemetry_sha256=force['sha256'], transitions_per_chain=250)))
    row = dict(tuning=dict(passed=True, manifest=str(tune_path)), chunks=chunks, retained_draws=2000,
        looks=[dict(draws_per_chain=n, status='INCONCLUSIVE', diagnostic_valid=False) for n in (1000, 2000)],
        status='INCONCLUSIVE')
    save(output/'report.json', dict(status='COMPLETED_BOUNDED_INFERENCE', model_id='SV20-T20', arms=dict(B=row),
        comparisons={'B/R': 'INCONCLUSIVE', 'C/D': 'NOT_RUN', 'D/R': 'NOT_RUN', 'C/R': 'NOT_RUN'}))
    return output


@pytest.mark.parametrize('nested', [False, True])
def test_preserved_attempt_is_reported_without_new_inference_authority(tmp_path, nested):
    output = preserved(tmp_path, nested)
    # An unrelated current-source edit must not erase archived evidence.
    (tmp_path/'runner.py').write_text('new reporting source')
    state = program.sv20_inference_history(tmp_path)
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
            v['config']['seed'] = [20260922, 36000]
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
    assert program.sv20_inference_history(tmp_path)['status'] == 'ARTIFACT_INVALID'


def test_failed_or_in_progress_attempt_is_preserved(tmp_path):
    output = preserved(tmp_path)
    path = output/'run-manifest.json'
    v = json.loads(path.read_text())
    del v['exit_code']
    save(path, v)
    assert program.sv20_inference_history(tmp_path)['status'] == 'RUNNING'
    v['exit_code'] = 1
    save(path, v)
    save(output/'report.json', dict(status='EXECUTION_FAILED', model_id='SV20-T20'))
    state = program.sv20_inference_history(tmp_path)
    assert state['status'] == 'EXECUTION_FAILED'
    assert state['authorizes_new_inference'] is False


@pytest.mark.parametrize('failure', ['missing', 'partial', 'continued_after_fallback', 'counts'])
def test_original_force_condition_cannot_be_omitted_or_ignored(tmp_path, failure):
    output = preserved(tmp_path)
    path = output/'report.json'
    report = json.loads(path.read_text())
    chunks = report['arms']['B']['chunks']
    if failure == 'missing':
        for chunk in chunks:
            del chunk['original_force_health']
    elif failure == 'partial':
        del chunks[-1]['original_force_health']
    else:
        record = chunks[-1]['original_force_health']
        record['force_fallback_count_by_chain'] = [1, 0, 0, 0]
        if failure == 'continued_after_fallback':
            record['passed'] = False
    save(path, report)
    assert program.sv20_inference_history(tmp_path)['status'] == 'ARTIFACT_INVALID'


def test_scope_force_veto_is_preserved_when_generic_archive_health_passes(tmp_path):
    output = preserved(tmp_path)
    path = output/'report.json'
    report = json.loads(path.read_text())
    row = report['arms']['B']
    row['chunks'][-1]['original_force_health'].update(
        passed=False, force_fallback_count_by_chain=[0, 1, 0, 0])
    row['looks'] = row['looks'][:1]
    row['retained_draws'] = 1750
    row['status'] = report['comparisons']['B/R'] = 'HEALTH_VETO'
    save(path, report)
    state = program.sv20_inference_history(tmp_path)
    assert state['artifact_valid'] is True
    assert state['comparisons']['B/R'] == 'HEALTH_VETO'
    assert state['original_force_health']['passed'] is False
    assert all(chunk['health_passed'] for chunk in row['chunks'])


def test_two_attempts_require_reconciliation(tmp_path):
    preserved(tmp_path)
    preserved(tmp_path, nested=True)
    assert program.sv20_inference_history(tmp_path)['status'] == 'RECONCILIATION_REQUIRED'


def test_master_launch_manifest_does_not_count_as_an_inference_attempt(tmp_path):
    output = tmp_path/'docs/plans/artifacts/surrogate-hmc-all-models-test-sv20-inference-01'
    save(output/'run-manifest.json', dict(schema='master-orchestration-test', scope='SV20', jobs=[],
        command_line=['python', 'scripts/run_surrogate_hmc_program.py']))
    assert program.sv20_inference_history(tmp_path)['status'] == 'NOT_RUN'
    preserved(tmp_path, nested=True)
    state = program.sv20_inference_history(tmp_path)
    assert state['artifact_valid'] is True
    assert len(state['attempt_manifests']) == 1


def test_master_dispatches_sv_runner_without_changing_frozen_inputs(tmp_path, monkeypatch):
    from bayesfilter.inference import surrogate_hmc_program as master
    from scripts import run_surrogate_hmc_sv as runner
    calls = []
    monkeypatch.setattr(runner, 'main', lambda args: calls.append(args) or 0)
    output = tmp_path/'run'
    assert master.main(['--stage', 'P3-SV20-inference', '--scope', 'SV20',
                        '--output', str(output), '--dry-run']) == 0
    assert calls == [['--output', str(output), '--dry-run']]
    with pytest.raises(SystemExit):
        master.main(['--stage', 'P3-SV20-inference', '--scope', 'SV20',
                     '--output', str(output), '--selection', 'unrelated.json'])


def test_actual_sv_tuning_uses_prior_chart_and_reserved_l1_schedule():
    from scripts.run_surrogate_hmc_sv import tuning_config
    config = tuning_config('SV20-test')
    assert config.nomination.max_leapfrog_steps == 1
    assert config.nomination.mass_window_results == (32, 64)
    assert config.warmup_results == 250 and config.verification_results == 1000
    assert config.initial_chain_positions == ((-2., -2.), (-.7, .7), (.7, -.7), (2., 2.))


def test_chunk_functionals_keep_physical_units_order_and_partial_batch(monkeypatch):
    import tensorflow as tf
    from bayesfilter.testing import surrogate_hmc_sv_inference_metrics as metrics
    calls = []
    def factory(**kwargs):
        assert kwargs == dict(intervals=1024, half_width=16., batch_size=32)
        def statistics(theta, observations):
            calls.append(theta.shape)
            return tf.zeros([32], tf.float64), theta, tf.zeros([32], tf.float64)
        return statistics
    monkeypatch.setattr(metrics, 'make_statistics', factory)
    reference = dict(means=[.5, .5], scales=[.2, .4], cuts=[[.2, .5, .8]]*2)
    evaluate = metrics.make_functionals(dict(raw_observations=[.3]*20), reference)
    samples = tf.reshape(tf.linspace(tf.constant(-1., tf.float64), tf.constant(1., tf.float64), 2000), [250, 4, 2])
    previous = tf.config.functions_run_eagerly()
    try:
        tf.config.run_functions_eagerly(True)
        parameters, functions, valid = evaluate(samples)
    finally:
        tf.config.run_functions_eagerly(previous)
    expected = .1+.8*.5*(1+tf.math.erf(samples/tf.sqrt(tf.constant(2., tf.float64))))
    assert bool(tf.reduce_all(tf.abs(parameters-expected) < 1e-14))
    assert len(calls) == 32 and all(shape == (32, 2) for shape in calls)
    assert functions.shape == (250, 4, 13) and valid.shape == (1000,)
    assert bool(tf.reduce_all(valid))
    assert bool(tf.reduce_all(functions[:, :, -2:] == parameters))
    assert bool(tf.reduce_all(tf.abs(functions[:, :, 0]-(parameters[:, :, 0]-.5)/.2) < 1e-14))
    assert functions[0, 0, 5:8].numpy().tolist() == [0., 1., 1.]
    assert functions[-1, -1, 5:8].numpy().tolist() == [0., 0., 1.]
