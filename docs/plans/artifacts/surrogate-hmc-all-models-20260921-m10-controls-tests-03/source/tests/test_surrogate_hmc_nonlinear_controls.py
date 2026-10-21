"""Control reuse, budget accounting and non-authoritative cost screening."""
import json
from pathlib import Path

import pytest

from bayesfilter.inference import surrogate_hmc_program as program
from bayesfilter.inference.surrogate_hmc_dctr_selection import SCHEMA, USE, MULTIPLIERS
from scripts.calibrate_surrogate_hmc_nonlinear_controls import optimistic_forecast


def measurements():
    return [dict(compilation_included=i == 0,wall_seconds=10. if i == 0 else 1.,
                 cpu_worker_seconds=5. if i == 0 else .5) for i in range(5)]


def test_forecast_preserves_per_arm_and_cpu_limit():
    result = optimistic_forecast(measurements(),dict(gpu_seconds=10000.,cpu_worker_seconds=1000.))
    assert result['status'] == 'UNDER_BUDGETED_INITIAL_ALLOCATION'
    assert result['optimistic_cost'] == dict(gpu_seconds=4890.,cpu_worker_seconds=2445.)
    assert result['authorizes_inference'] is False
    result = optimistic_forecast(measurements(),dict(gpu_seconds=10000.,cpu_worker_seconds=10000.))
    assert result['status'] == 'REQUIRES_COMPLETE_FORECAST'
    assert result['authorizes_inference'] is False


@pytest.mark.parametrize('bad', [0.,-1.,float('nan'),float('inf')])
def test_invalid_timing_cannot_nominate_a_launch(bad):
    records = measurements()
    records[2]['cpu_worker_seconds'] = bad
    with pytest.raises(ValueError,match='positive measured'):
        optimistic_forecast(records,dict(gpu_seconds=10000.,cpu_worker_seconds=10000.))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return dict(path=str(path), sha256=program.file_hash(path))


def control_evidence(root, monkeypatch):
    """Synthetic historical evidence exercises the real reuse validator."""
    output = root/'docs/plans/artifacts/surrogate-hmc-all-models-test-m10-controls-01'
    sources = {}
    for name in ('bayesfilter/inference/surrogate_hmc_dctr_selection.py',
                 'bayesfilter/testing/surrogate_hmc_nonlinear_consumer.py'):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen numerical source')
        sources[name] = program.file_hash(path)
    scope = dict(model='M10', attempt=1, horizon=2, particle_count=1024,
        dtype='float32', tf32=True, jit_compile=True, state_dimension=1, parameter_dimension=2,
        reset='contract_e_chol_v1', chunk_policy='dpf_transport_exact_divisor_cap3000_v1', chunk_size=1024)
    selection = dict(schema=SCHEMA, use=USE, exact_score_admission=False, default_readiness=False,
        scope=scope, source_hashes=sources, selected_index=1, selected_multiplier=.1,
        controls={'test_setting': .1}, evidence={}, partition_data={},
        issuer_sha256=sources['bayesfilter/inference/surrogate_hmc_dctr_selection.py'])
    for partition in ('calibration', 'validation'):
        original, data = {}, {}
        for regime in ('nominal', 'stress'):
            data[regime] = dict(role=partition, regime=regime, horizon=2, dtype='float32')
            original[regime] = save(output/partition/(regime+'.json'), data[regime])
        selection['partition_data'][partition] = save(output/partition/'data.json',
            dict(model_id='M10', attempt=1, partition=partition, original_datasets=original, data=data))
        arms = [save(output/partition/f'arm-{i}.json',
            dict(multiplier=m, controls={'test_setting': m})) for i,m in enumerate(MULTIPLIERS)]
        pairs = [dict(non_harm_passed=i > 0, healthy_nominal=True, stress_safe=True,
            no_new_stress_failure=True, normalized_maximum_change=.01 if i == 0 else .0001,
            tolerance=.001) for i in range(3)]
        selection['evidence'][partition] = save(output/partition/'curve.json',
            dict(scope=scope, source_hashes=sources, arms=arms, pairs=pairs,
                 data_sha256=selection['partition_data'][partition]['sha256']))
    save(output/'selection.json', selection)
    save(output/'report.json', dict(status='SELECTED_FOR_PROPOSAL_EXPERIMENT', selected=selection,
                                   scope=scope, forecast=dict(authorizes_inference=False)))
    save(output/'run-manifest.json', dict(source_hashes=sources,
        command=['python', 'scripts/calibrate_surrogate_hmc_nonlinear_controls.py']))
    save(output/'accounting.json', dict(charged_this_run=dict(cpu_worker_seconds=10., gpu_seconds=20.)))
    monkeypatch.setattr(program, 'm10_consumer_state', lambda root: dict(
        family_remaining=dict(cpu_worker_seconds=500., gpu_seconds=6000.)))
    return output


def test_control_reuse_preserves_preparation_only_status(tmp_path, monkeypatch):
    output = control_evidence(tmp_path, monkeypatch)
    state = program.m10_controls_state(tmp_path)
    assert state['qualified']['report'] == str(output/'report.json')
    assert state['qualified']['authorizes_new_inference'] is False
    assert state['cpu_worker_seconds'] == 340.
    assert state['wall_seconds'] == 430.


@pytest.mark.parametrize('target, message', [
    ('bayesfilter/testing/surrogate_hmc_nonlinear_consumer.py', 'numerical source changed'),
    ('calibration/nominal.json', 'original control data changed'),
    ('validation/arm-1.json', 'control arm changed'),
    ('calibration/curve.json', 'control evidence changed'),
])
def test_control_reuse_rejects_source_and_evidence_drift(tmp_path, monkeypatch, target, message):
    output = control_evidence(tmp_path, monkeypatch)
    path = (tmp_path if target.startswith('bayesfilter/') else output)/target
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match=message):
        program.m10_controls_state(tmp_path)


def test_control_reuse_rejects_unmeasured_selection(tmp_path, monkeypatch):
    output = control_evidence(tmp_path, monkeypatch)
    report = json.loads((output/'report.json').read_text())
    report['selected']['controls'] = {'test_setting': 999.}
    save(output/'selection.json', report['selected'])
    save(output/'report.json', report)
    with pytest.raises(ValueError, match='differ from measured arm'):
        program.m10_controls_state(tmp_path)


def test_mixed_master_control_accounting_is_counted_once(tmp_path, monkeypatch):
    output = control_evidence(tmp_path, monkeypatch)
    nested = output.parent/'surrogate-hmc-all-models-mixed/P3-M10-controls'
    save(nested/'accounting.json', dict(charged_this_run=dict(cpu_worker_seconds=8., gpu_seconds=9.)))
    save(nested.parent/'accounting.json', dict(jobs=[dict(
        command=['python', 'scripts/calibrate_surrogate_hmc_nonlinear_controls.py', '--output', str(nested)],
        cpu_worker_seconds=8., gpu_seconds=9.)]))
    history = program.m10_control_attempts(tmp_path)
    assert history['spent'] == dict(cpu_worker_seconds=18., gpu_seconds=29.)
    assert len(history['reports']) == 2
    with pytest.raises(ValueError, match='missing M10 control report'):
        program.m10_controls_state(tmp_path)


def test_master_control_dependency_does_not_require_posterior_reference(tmp_path, monkeypatch, capsys):
    state = dict(wall_seconds=50., cpu_worker_seconds=40., qualified=None,
                 family_remaining=dict(cpu_worker_seconds=500., gpu_seconds=6000.))
    for name in ('m10_reference_state', 'm10_consumer_state', 'm10_posterior_state', 'm10_controls_state'):
        monkeypatch.setattr(program, name, lambda root: dict(state))
    assert program.main(['--stage', 'all', '--scope', 'M10',
                         '--output', str(tmp_path/'unused'), '--dry-run']) == 0
    launch = json.loads(capsys.readouterr().out)
    job = next(j for j in launch['jobs'] if j['name'] == 'P3-M10-controls-job')
    assert job['depends_on'] == ['P3-M10-qualify-job']
    assert '--qualification-root' in job['command']
    assert '--family-cpu-budget' in job['command']
    assert launch['scope_cards']['M10']['surrogate_hmc_tested'] is False
